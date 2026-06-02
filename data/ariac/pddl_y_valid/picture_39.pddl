(define (problem picture_39)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        green_pump red_battery green_regulator blue_battery - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear green_pump)
        (clear red_battery)
        (clear green_regulator)
        (clear blue_battery)
        (part_at green_pump table)
        (part_at red_battery table)
        (part_at green_regulator table)
        (part_at blue_battery table)
    )
    
    (:goal
        (and
            (part_at green_pump pump_placement)
            (part_at green_regulator regulator_placement)
            (part_at red_battery battery_placement)
        )
    )
)
