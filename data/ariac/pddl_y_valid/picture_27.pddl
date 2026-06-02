(define (problem picture_27)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        red_pump green_battery green_regulator - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear green_battery)
        (clear green_regulator)
        (part_at red_pump table)
        (on green_battery red_pump)
        (part_at green_regulator table)
    )
    
    (:goal
        (and
            (part_at red_pump pump_placement)
            (part_at green_regulator regulator_placement)
            (part_at green_battery battery_placement)
        )
    )
)
