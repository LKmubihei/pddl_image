(define (problem picture_7)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        blue_pump green_battery green_regulator - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear blue_pump)
        (clear green_battery)
        (clear green_regulator)
        (part_at green_regulator table)
        (part_at blue_pump table)
        (part_at green_battery table)
    )
    
    (:goal
        (and
            (part_at blue_pump pump_placement)
            (part_at green_regulator regulator_placement)
            (part_at green_battery battery_placement)
        )
    )
)
